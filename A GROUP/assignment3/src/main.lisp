;;;; main.lisp — orchestration: build datasets, train the three models,
;;;; evaluate under two protocols, and write results.
;;;;
;;;; Protocol A — fixed 80/20 temporal split (as in the Assignment 2
;;;;   write-up): train on the first 80% of months, test on the last 20%.
;;;; Protocol B — walk-forward validation (standard for time series):
;;;;   for each test month t, retrain on all months < t and predict t.

(in-package :cpp4103)

;;; ------------------------------------------------------------------
;;; Evaluation metrics
;;; ------------------------------------------------------------------

(defun rmse (actual predicted)
  (sqrt (/ (loop for a across actual for p across predicted
                 sum (expt (- a p) 2))
           (length actual))))

(defun mae (actual predicted)
  (/ (loop for a across actual for p across predicted
           sum (abs (- a p)))
     (length actual)))

(defun r-squared (actual predicted)
  (let* ((mu (vmean actual))
         (ss-res (loop for a across actual for p across predicted
                       sum (expt (- a p) 2)))
         (ss-tot (loop for a across actual sum (expt (- a mu) 2))))
    (if (zerop ss-tot) 0d0 (- 1d0 (/ ss-res ss-tot)))))

(defun mape (actual predicted)
  (* 100d0
     (/ (loop for a across actual for p across predicted
              sum (/ (abs (- a p)) (max (abs a) 1d-9)))
        (length actual))))

;;; ------------------------------------------------------------------
;;; Array helpers
;;; ------------------------------------------------------------------

(defun slice-rows (x from to)
  (let* ((p (array-dimension x 1))
         (out (make-array (list (- to from) p) :element-type 'double-float)))
    (loop for i from from below to do
      (loop for j below p do
        (setf (aref out (- i from) j) (aref x i j))))
    out))

(defun slice-vec (y from to)
  (let ((out (make-array (- to from) :element-type 'double-float)))
    (loop for i from from below to do (setf (aref out (- i from)) (aref y i)))
    out))

(defun row-as-vector (x i)
  (let* ((p (array-dimension x 1))
         (v (make-array p :element-type 'double-float)))
    (dotimes (j p) (setf (aref v j) (aref x i j)))
    v))

(defun build-sequences (xs indices)
  "3-month lookback sequences (rows t-2, t-1, t of XS) for each row index."
  (loop for i in indices
        collect (list (row-as-vector xs (- i 2))
                      (row-as-vector xs (- i 1))
                      (row-as-vector xs i))))

(defun standardize-y (y n-train)
  (let* ((tr (slice-vec y 0 n-train))
         (mu (vmean tr))
         (sd (let ((s (vstd tr))) (if (< s 1d-9) 1d0 s)))
         (out (make-array (length y) :element-type 'double-float)))
    (dotimes (i (length y))
      (setf (aref out i) (/ (- (aref y i) mu) sd)))
    (values out mu sd)))

;;; ------------------------------------------------------------------
;;; Unified train/predict for the three models
;;; ------------------------------------------------------------------

(defparameter *lstm-hidden* 8)
(defparameter *lstm-epochs* 150)
(defparameter *lstm-lr* 0.01d0)
(defparameter *ridge-l2* 1.0d0)

(defun slice-rows-by-index (x indices)
  (let* ((p (array-dimension x 1))
         (out (make-array (list (length indices) p) :element-type 'double-float)))
    (loop for i in indices for k from 0
          do (loop for j below p do (setf (aref out k j) (aref x i j))))
    out))

(defun fit-predict (kind x y train-end test-indices)
  "Standardize on rows [0, TRAIN-END), fit model KIND, predict rows in
   TEST-INDICES. Returns (values predictions model)."
  (let* ((scaler (fit-scaler x train-end))
         (xs (transform-x x scaler))
         (x-train (slice-rows xs 0 train-end))
         (y-train (slice-vec y 0 train-end))
         (n-test (length test-indices))
         (preds (make-array n-test :element-type 'double-float))
         (model nil))
    (ecase kind
      (:linreg
       (setf model (linreg-fit x-train y-train :l2 *ridge-l2*))
       (loop for i in test-indices for k from 0
             do (setf (aref preds k) (linreg-predict-row model xs i))))
      (:forest
       (setf model (forest-fit x-train y-train
                               :n-trees 200 :max-depth 10 :min-leaf 2
                               :feature-names *feature-names*))
       (let ((p (forest-predict model
                                (slice-rows-by-index xs test-indices))))
         (loop for k below n-test do (setf (aref preds k) (aref p k)))))
      (:lstm
       (multiple-value-bind (ys y-mu y-sd) (standardize-y y train-end)
         (let* ((train-idx (loop for i from 2 below train-end collect i))
                (seq-train (build-sequences xs train-idx))
                (yt (make-array (length train-idx) :element-type 'double-float)))
           (loop for i in train-idx for k from 0
                 do (setf (aref yt k) (aref ys i)))
           (setf model (lstm-fit seq-train yt
                                 :h *lstm-hidden* :epochs *lstm-epochs*
                                 :lr *lstm-lr*))
           (let ((p (lstm-predict model (build-sequences xs test-indices))))
             (loop for k below n-test
                   do (setf (aref preds k) (+ (* (aref p k) y-sd) y-mu))))))))
    (values preds model)))

(defun walk-forward (kind x y test-indices)
  "Expanding-window evaluation: for each index t, train on [0, t) and
   predict t. Returns the vector of one-step-ahead predictions."
  (let ((preds (make-array (length test-indices) :element-type 'double-float)))
    (loop for i in test-indices for k from 0
          do (setf (aref preds k) (aref (fit-predict kind x y i (list i)) 0)))
    preds))

;;; ------------------------------------------------------------------
;;; Per-commodity experiment
;;; ------------------------------------------------------------------

(defstruct result commodity model protocol rmse mae r2 mape predictions)

(defun evaluate (commodity model-name protocol actual predicted)
  (make-result :commodity commodity :model model-name :protocol protocol
               :rmse (rmse actual predicted)
               :mae (mae actual predicted)
               :r2 (r-squared actual predicted)
               :mape (mape actual predicted)
               :predictions predicted))

(defparameter *model-kinds*
  '((:linreg . "Linear Regression")
    (:forest . "Random Forest")
    (:lstm . "LSTM")))

(defun run-experiment (ds)
  "Returns (values results test-months y-test rf-model)."
  (let* ((x (dataset-x ds))
         (y (dataset-y ds))
         (n (array-dimension x 0))
         (n-train (floor (* n 0.8d0)))
         (test-indices (loop for i from n-train below n collect i))
         (y-test (slice-vec y n-train n))
         (name (dataset-name ds))
         (results '())
         (rf-model nil))
    (format t "~%=== ~a ===~%" name)
    (format t "    samples: ~d train / ~d test (strict temporal ordering)~%"
            n-train (- n n-train))
    (format t "    train: ~a .. ~a   test: ~a .. ~a~%"
            (nth 0 (dataset-months ds)) (nth (1- n-train) (dataset-months ds))
            (nth n-train (dataset-months ds)) (nth (1- n) (dataset-months ds)))
    ;; Naive persistence baseline: predict last month's observed price.
    ;; Identical under both protocols; reported under each for comparison.
    (let ((naive (make-array (length test-indices) :element-type 'double-float)))
      (loop for i in test-indices for k from 0
            do (setf (aref naive k) (aref y (1- i))))
      (push (evaluate name "Naive (previous month)" "fixed-split" y-test naive) results)
      (push (evaluate name "Naive (previous month)" "walk-forward" y-test naive) results))
    (dolist (mk *model-kinds*)
      (destructuring-bind (kind . mname) mk
        ;; Protocol A: fixed split
        (multiple-value-bind (preds model)
            (fit-predict kind x y n-train test-indices)
          (when (eq kind :forest) (setf rf-model model))
          (push (evaluate name mname "fixed-split" y-test preds) results))
        ;; Protocol B: walk-forward
        (let ((preds (walk-forward kind x y test-indices)))
          (push (evaluate name mname "walk-forward" y-test preds) results))))
    (values (nreverse results)
            (subseq (dataset-months ds) n-train)
            y-test
            rf-model)))

;;; ------------------------------------------------------------------
;;; Reporting
;;; ------------------------------------------------------------------

(defun print-results-table (results protocol title)
  (format t "~%    ~a~%" title)
  (format t "    ~22a ~10a ~9a ~8a ~7a~%" "Model" "RMSE(KES)" "MAE(KES)" "R2" "MAPE%")
  (format t "    ~a~%" (make-string 62 :initial-element #\-))
  (dolist (r results)
    (when (string= (result-protocol r) protocol)
      (format t "    ~22a ~10,2f ~9,2f ~8,3f ~7,1f~%"
              (result-model r) (result-rmse r) (result-mae r)
              (result-r2 r) (result-mape r)))))

(defun write-metrics-csv (all-results)
  (with-open-file (out (results-path "metrics.csv")
                       :direction :output :if-exists :supersede)
    (format out "commodity,model,protocol,rmse_kes,mae_kes,r_squared,mape_pct~%")
    (dolist (r all-results)
      (format out "~a,~a,~a,~,3f,~,3f,~,4f,~,2f~%"
              (result-commodity r) (result-model r) (result-protocol r)
              (result-rmse r) (result-mae r) (result-r2 r) (result-mape r)))))

(defun write-predictions-csv (fname months actual results)
  (let ((wf (remove-if-not (lambda (r) (string= (result-protocol r) "walk-forward"))
                           results)))
    (with-open-file (out (results-path fname)
                         :direction :output :if-exists :supersede)
      (format out "month,actual~{,~a~}~%"
              (mapcar (lambda (r) (substitute #\_ #\Space (result-model r))) wf))
      (loop for m in months for i from 0 do
        (format out "~a,~,2f~{,~,2f~}~%"
                m (aref actual i)
                (mapcar (lambda (r) (aref (result-predictions r) i)) wf))))))

(defun write-importance-csv (fname model)
  (with-open-file (out (results-path fname)
                       :direction :output :if-exists :supersede)
    (format out "feature,importance~%")
    (let* ((imp (forest-importance model))
           (names (forest-feature-names model))
           (pairs (sort (loop for nm in names for j from 0
                              collect (cons nm (aref imp j)))
                        #'> :key #'cdr)))
      (dolist (p pairs)
        (format out "~a,~,4f~%" (car p) (cdr p))))))

(defun safe-name (commodity)
  (string-downcase
   (with-output-to-string (s)
     (loop for ch across commodity
           do (cond ((alphanumericp ch) (write-char ch s))
                    ((char= ch #\Space) (write-char #\_ s)))))))

(defun write-series-csv (raw)
  "Dump the assembled monthly series grid for plotting/inspection."
  (with-open-file (out (results-path "series.csv")
                       :direction :output :if-exists :supersede)
    (format out "month,diesel,petrol,kerosene,usd_kes,cpi,inflation_yoy~{,~a~}~%"
            (mapcar (lambda (tg) (safe-name (car tg))) (raw-data-targets raw)))
    (loop for m in (raw-data-months raw) for i from 0 do
      (format out "~a,~,2f,~,2f,~,2f,~,2f,~,2f,~,2f~{,~,2f~}~%"
              m
              (aref (raw-data-diesel raw) i)
              (aref (raw-data-petrol raw) i)
              (aref (raw-data-kerosene raw) i)
              (aref (raw-data-fx raw) i)
              (aref (raw-data-cpi raw) i)
              (aref (raw-data-inflation raw) i)
              (mapcar (lambda (tg) (aref (cdr tg) i)) (raw-data-targets raw))))))

(defun print-consolidated (all-results protocol title)
  (format t "~%~a~%" title)
  (format t "~22a ~12a ~8a ~9a~%" "Model" "Avg RMSE" "Avg R2" "Avg MAPE%")
  (dolist (mname (cons "Naive (previous month)" (mapcar #'cdr *model-kinds*)))
    (let ((rs (remove-if-not
               (lambda (r) (and (string= (result-model r) mname)
                                (string= (result-protocol r) protocol)))
               all-results)))
      (format t "~22a ~12,2f ~8,3f ~9,1f~%"
              mname
              (/ (reduce #'+ (mapcar #'result-rmse rs)) (length rs))
              (/ (reduce #'+ (mapcar #'result-r2 rs)) (length rs))
              (/ (reduce #'+ (mapcar #'result-mape rs)) (length rs))))))

;;; ------------------------------------------------------------------
;;; Entry point
;;; ------------------------------------------------------------------

(defun run ()
  (format t "~%CPP 4103 Assignment 3 — Group A~%")
  (format t "Predicting Kenyan commodity prices from fuel prices and~%")
  (format t "macroeconomic indicators — Common Lisp implementation~%")
  (format t "~a~%" (make-string 70 :initial-element #\=))
  (let ((raw (load-raw-data)))
    (format t "~%Data loaded: ~d monthly observations, ~a .. ~a (Nairobi)~%"
            (length (raw-data-months raw))
            (first (raw-data-months raw))
            (car (last (raw-data-months raw))))
    (format t "~%Exploratory check — Pearson correlation with diesel price:~%")
    (dolist (target (raw-data-targets raw))
      (format t "    ~35a r = ~,3f~%"
              (car target) (pearson (raw-data-diesel raw) (cdr target))))
    (write-series-csv raw)
    (let ((all-results '()))
      (dolist (target (raw-data-targets raw))
        (let ((ds (build-dataset raw (car target) (cdr target))))
          (multiple-value-bind (results months y-test rf-model) (run-experiment ds)
            (print-results-table results "fixed-split"
                                 "Protocol A — fixed 80/20 temporal split")
            (print-results-table results "walk-forward"
                                 "Protocol B — walk-forward (retrain each month)")
            (setf all-results (append all-results results))
            (let ((base (safe-name (car target))))
              (write-predictions-csv (format nil "predictions_~a.csv" base)
                                     months y-test results)
              (write-importance-csv (format nil "importance_~a.csv" base)
                                    rf-model))
            (let* ((imp (forest-importance rf-model))
                   (names (forest-feature-names rf-model))
                   (pairs (sort (loop for nm in names for j from 0
                                      collect (cons nm (aref imp j)))
                                #'> :key #'cdr)))
              (format t "~%    Top Random Forest features:~%")
              (loop for p in pairs repeat 5
                    do (format t "      ~25a ~,3f~%" (car p) (cdr p)))))))
      (write-metrics-csv all-results)
      (format t "~%~a" (make-string 70 :initial-element #\=))
      (print-consolidated all-results "fixed-split"
                          "CONSOLIDATED — Protocol A (fixed 80/20 split)")
      (print-consolidated all-results "walk-forward"
                          "CONSOLIDATED — Protocol B (walk-forward)")
      (format t "~%Results written to results/ (metrics.csv, predictions_*.csv, importance_*.csv)~%")))
  t)
