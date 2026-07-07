;;;; dataset.lisp — load Kenya price data and engineer model features
;;;; Data sources (all real, downloaded):
;;;;   data/wfp_food_prices_ken.csv      — WFP via Humanitarian Data Exchange
;;;;   data/worldbank_fx_usdkes.csv      — World Bank GEM, monthly USD/KES
;;;;   data/worldbank_cpi.csv            — World Bank GEM, monthly CPI level
;;;;   data/worldbank_inflation_yoy.csv  — World Bank GEM, monthly y/y inflation %

(in-package :cpp4103)

(defvar *base-dir* nil "Project root; set by run.lisp")

(defun data-path (name)
  (merge-pathnames (concatenate 'string "data/" name) *base-dir*))

(defun results-path (name)
  (merge-pathnames (concatenate 'string "results/" name) *base-dir*))

;;; Study window: overlap of Nairobi fuel and commodity series
(defparameter *grid* (month-range "2014-07" "2020-12"))

;;; ------------------------------------------------------------------
;;; WFP Nairobi monthly series
;;; ------------------------------------------------------------------

(defun load-wfp-table ()
  "Read the WFP Kenya CSV once. Returns (values header rows)."
  (read-csv (data-path "wfp_food_prices_ken.csv")))

(defun wfp-series (header rows commodity unit)
  "Monthly Nairobi price series for COMMODITY/UNIT.
   Multiple observations within a month are averaged.
   Returns a hash: \"YYYY-MM\" -> mean price (KES)."
  (let ((c-date (position "date" header :test #'string=))
        (c-market (position "market" header :test #'string=))
        (c-comm (position "commodity" header :test #'string=))
        (c-unit (position "unit" header :test #'string=))
        (c-price (position "price" header :test #'string=))
        (acc (make-hash-table :test #'equal)))
    (dolist (row rows)
      (when (and (string= (nth c-market row) "Nairobi")
                 (string= (nth c-comm row) commodity)
                 (string= (nth c-unit row) unit))
        (let ((price (parse-num (nth c-price row)))
              (mkey (subseq (nth c-date row) 0 7)))
          (when price
            (push price (gethash mkey acc))))))
    (let ((out (make-hash-table :test #'equal)))
      (maphash (lambda (k v)
                 (setf (gethash k out) (/ (reduce #'+ v) (length v))))
               acc)
      out)))

(defun load-worldbank-series (filename)
  "Load a two-column (month,value) CSV into a hash \"YYYY-MM\" -> value."
  (multiple-value-bind (header rows) (read-csv (data-path filename))
    (declare (ignore header))
    (let ((out (make-hash-table :test #'equal)))
      (dolist (row rows)
        (let ((v (parse-num (second row))))
          (when v (setf (gethash (first row) out) v))))
      out)))

;;; ------------------------------------------------------------------
;;; Assemble all raw series on the monthly grid
;;; ------------------------------------------------------------------

(defstruct raw-data
  months     ; list of "YYYY-MM"
  diesel petrol kerosene   ; KES per litre, Nairobi pump prices
  fx cpi inflation         ; World Bank monthly Kenya series
  targets)                 ; alist (name . price-vector)

(defun load-raw-data ()
  (multiple-value-bind (header rows) (load-wfp-table)
    (flet ((series (commodity unit)
             (interpolate-series *grid* (wfp-series header rows commodity unit))))
      (make-raw-data
       :months *grid*
       :diesel (series "Fuel (diesel)" "L")
       :petrol (series "Fuel (petrol-gasoline)" "L")
       :kerosene (series "Fuel (kerosene)" "L")
       :fx (interpolate-series *grid* (load-worldbank-series "worldbank_fx_usdkes.csv"))
       :cpi (interpolate-series *grid* (load-worldbank-series "worldbank_cpi.csv"))
       :inflation (interpolate-series *grid* (load-worldbank-series "worldbank_inflation_yoy.csv"))
       :targets
       (list (cons "Maize (KG, wholesale)" (series "Maize" "KG"))
             (cons "Vegetable oil (1L, retail)" (series "Oil (vegetable)" "L"))
             (cons "Bread (400g, retail)" (series "Bread" "400 G")))))))

;;; ------------------------------------------------------------------
;;; Feature engineering (mirrors the Assignment 2 methodology)
;;; ------------------------------------------------------------------

(defparameter *feature-names*
  '("diesel" "petrol" "kerosene" "fuel_composite" "petrol_diesel_ratio"
    "diesel_lag1" "diesel_lag2" "diesel_lag3" "fuel_composite_lag1"
    "diesel_mom_pct" "cpi" "cpi_mom_pct" "inflation_yoy" "usd_kes"
    "target_lag1" "target_lag2" "target_lag3" "month_sin" "month_cos"))

(defstruct dataset
  name months x y   ; X: n x p double array, Y: length-n vector
  feature-names)

(defun build-dataset (raw target-name target)
  "Supervised rows for months with 3 months of history available.
   Row t predicts the commodity price in month t from current fuel and
   macro conditions plus lagged prices (delayed transmission effects)."
  (let* ((months (raw-data-months raw))
         (n-grid (length months))
         (diesel (raw-data-diesel raw))
         (petrol (raw-data-petrol raw))
         (kerosene (raw-data-kerosene raw))
         (fx (raw-data-fx raw))
         (cpi (raw-data-cpi raw))
         (infl (raw-data-inflation raw))
         (start 3)
         (n (- n-grid start))
         (p (length *feature-names*))
         (x (make-array (list n p) :element-type 'double-float))
         (y (make-array n :element-type 'double-float))
         (kept-months '()))
    (loop for time from start below n-grid
          for i from 0
          do (let* ((fuel-comp (lambda (k) (/ (+ (aref petrol k) (aref diesel k)) 2d0)))
                    (m (month-of (nth time months)))
                    (row (list
                          (aref diesel time)
                          (aref petrol time)
                          (aref kerosene time)
                          (funcall fuel-comp time)
                          (/ (aref petrol time) (aref diesel time))
                          (aref diesel (- time 1))
                          (aref diesel (- time 2))
                          (aref diesel (- time 3))
                          (funcall fuel-comp (1- time))
                          (* 100d0 (- (/ (aref diesel time) (aref diesel (1- time))) 1d0))
                          (aref cpi time)
                          (* 100d0 (- (/ (aref cpi time) (aref cpi (1- time))) 1d0))
                          (aref infl time)
                          (aref fx time)
                          (aref target (- time 1))
                          (aref target (- time 2))
                          (aref target (- time 3))
                          (sin (/ (* 2 pi m) 12d0))
                          (cos (/ (* 2 pi m) 12d0)))))
               (loop for v in row for j from 0
                     do (setf (aref x i j) (float v 1d0)))
               (setf (aref y i) (aref target time))
               (push (nth time months) kept-months)))
    (make-dataset :name target-name
                  :months (nreverse kept-months)
                  :x x :y y
                  :feature-names *feature-names*)))

;;; ------------------------------------------------------------------
;;; Temporal split and standardization
;;; ------------------------------------------------------------------

(defun temporal-split-index (n &optional (train-frac 0.8d0))
  (floor (* n train-frac)))

(defstruct scaler mu sd)

(defun fit-scaler (x n-train)
  "Column means/stds computed on the first N-TRAIN rows only."
  (let* ((p (array-dimension x 1))
         (mu (make-array p :element-type 'double-float))
         (sd (make-array p :element-type 'double-float)))
    (loop for j below p do
      (let ((col (make-array n-train :element-type 'double-float)))
        (loop for i below n-train do (setf (aref col i) (aref x i j)))
        (setf (aref mu j) (vmean col))
        (let ((s (vstd col)))
          (setf (aref sd j) (if (< s 1d-9) 1d0 s)))))
    (make-scaler :mu mu :sd sd)))

(defun transform-x (x scaler)
  (let* ((n (array-dimension x 0))
         (p (array-dimension x 1))
         (out (make-array (list n p) :element-type 'double-float))
         (mu (scaler-mu scaler))
         (sd (scaler-sd scaler)))
    (loop for i below n do
      (loop for j below p do
        (setf (aref out i j) (/ (- (aref x i j) (aref mu j)) (aref sd j)))))
    out))
