;;;; utils.lisp — CSV parsing, month arithmetic, statistics, linear algebra
;;;; CPP 4103 Assignment 3 — Group A
;;;; Pure Common Lisp (SBCL), no external libraries.

(defpackage :cpp4103
  (:use :common-lisp)
  (:export :run))

(in-package :cpp4103)

;;; Deterministic randomness so every run reproduces the same results
(defvar *rng* (sb-ext:seed-random-state 42))

(defun rand-int (n) (random n *rng*))
(defun rand-uniform (lo hi) (+ lo (* (- hi lo) (random 1d0 *rng*))))

;;; ------------------------------------------------------------------
;;; CSV parsing (handles double-quoted fields containing commas)
;;; ------------------------------------------------------------------

(defun strip-cr (s)
  (string-right-trim '(#\Return) s))

(defun parse-csv-line (line)
  "Split one CSV line into a list of string fields, honoring quotes."
  (let ((fields '())
        (cur (make-string-output-stream))
        (in-quotes nil)
        (n (length line))
        (i 0))
    (loop while (< i n) do
      (let ((ch (char line i)))
        (cond
          (in-quotes
           (cond ((char= ch #\")
                  (if (and (< (1+ i) n) (char= (char line (1+ i)) #\"))
                      (progn (write-char #\" cur) (incf i))
                      (setf in-quotes nil)))
                 (t (write-char ch cur))))
          ((char= ch #\") (setf in-quotes t))
          ((char= ch #\,) (push (get-output-stream-string cur) fields))
          (t (write-char ch cur))))
      (incf i))
    (push (get-output-stream-string cur) fields)
    (nreverse fields)))

(defun read-csv (path)
  "Read a CSV file. Returns (values header-list row-lists)."
  (with-open-file (in path :direction :input :external-format :utf-8)
    (let ((header nil) (rows '()))
      (loop for line = (read-line in nil nil)
            while line
            do (let ((line (strip-cr line)))
                 (unless (zerop (length line))
                   (if header
                       (push (parse-csv-line line) rows)
                       (setf header (parse-csv-line line))))))
      (values header (nreverse rows)))))

(defun parse-num (s)
  "Parse a numeric string to double-float, or NIL if empty/invalid."
  (when (and s (plusp (length s)))
    (let ((v (with-input-from-string (in s) (read in nil nil))))
      (when (numberp v) (float v 1d0)))))

;;; ------------------------------------------------------------------
;;; Month arithmetic — months keyed as "YYYY-MM"
;;; ------------------------------------------------------------------

(defun month-key->ordinal (key)
  "\"2014-07\" -> 2014*12 + 6"
  (let ((y (parse-integer key :start 0 :end 4))
        (m (parse-integer key :start 5 :end 7)))
    (+ (* y 12) (1- m))))

(defun ordinal->month-key (ord)
  (multiple-value-bind (y m) (floor ord 12)
    (format nil "~4,'0d-~2,'0d" y (1+ m))))

(defun month-range (start-key end-key)
  "List of \"YYYY-MM\" keys from start to end inclusive."
  (loop for o from (month-key->ordinal start-key)
          to (month-key->ordinal end-key)
        collect (ordinal->month-key o)))

(defun month-of (key)
  "Month number 1..12 from a \"YYYY-MM\" key."
  (parse-integer key :start 5 :end 7))

;;; ------------------------------------------------------------------
;;; Statistics
;;; ------------------------------------------------------------------

(defun vmean (v)
  (/ (loop for x across v sum x) (length v)))

(defun vstd (v)
  (let* ((mu (vmean v))
         (n (length v))
         (ss (loop for x across v sum (expt (- x mu) 2))))
    (if (<= n 1) 1d0 (sqrt (/ ss (1- n))))))

(defun pearson (a b)
  "Pearson correlation of two equal-length vectors."
  (let* ((ma (vmean a)) (mb (vmean b))
         (num 0d0) (da 0d0) (db 0d0))
    (loop for x across a for y across b
          do (incf num (* (- x ma) (- y mb)))
             (incf da (expt (- x ma) 2))
             (incf db (expt (- y mb) 2)))
    (if (or (zerop da) (zerop db)) 0d0 (/ num (sqrt (* da db))))))

(defun interpolate-series (grid table)
  "Build a double-float vector over GRID (list of month keys) from TABLE
   (hash month-key -> value). Interior gaps are linearly interpolated;
   edge gaps take the nearest known value."
  (let* ((n (length grid))
         (raw (make-array n :initial-element nil)))
    (loop for key in grid for i from 0
          do (setf (aref raw i) (gethash key table)))
    ;; collect known indices
    (let ((known (loop for i below n when (aref raw i) collect i)))
      (when (null known)
        (error "Series has no data on the grid"))
      (let ((out (make-array n :element-type 'double-float :initial-element 0d0)))
        (loop for i below n do
          (setf (aref out i)
                (cond
                  ((aref raw i) (float (aref raw i) 1d0))
                  ((< i (first known)) (float (aref raw (first known)) 1d0))
                  ((> i (car (last known))) (float (aref raw (car (last known))) 1d0))
                  (t ;; linear interpolation between neighbors
                   (let* ((lo (loop for k in known when (< k i) maximize k))
                          (hi (loop for k in known when (> k i) minimize k))
                          (frac (/ (- i lo) (float (- hi lo) 1d0))))
                     (+ (* (- 1d0 frac) (aref raw lo))
                        (* frac (aref raw hi))))))))
        out))))

;;; ------------------------------------------------------------------
;;; Linear algebra (dense, double-float 2D arrays)
;;; ------------------------------------------------------------------

(defun solve-linear-system (a b)
  "Solve A x = b by Gaussian elimination with partial pivoting.
   A is an n x n 2D array (destroyed), B a length-n vector (destroyed).
   Returns the solution vector."
  (let ((n (array-dimension a 0)))
    (loop for col below n do
      ;; pivot
      (let ((piv col))
        (loop for r from (1+ col) below n
              when (> (abs (aref a r col)) (abs (aref a piv col)))
                do (setf piv r))
        (when (/= piv col)
          (loop for j below n
                do (rotatef (aref a col j) (aref a piv j)))
          (rotatef (aref b col) (aref b piv))))
      (let ((d (aref a col col)))
        (when (< (abs d) 1d-12)
          (setf d (if (minusp d) -1d-12 1d-12)))
        (loop for r from (1+ col) below n do
          (let ((factor (/ (aref a r col) d)))
            (unless (zerop factor)
              (loop for j from col below n
                    do (decf (aref a r j) (* factor (aref a col j))))
              (decf (aref b r) (* factor (aref b col))))))))
    ;; back substitution
    (let ((x (make-array n :element-type 'double-float :initial-element 0d0)))
      (loop for r from (1- n) downto 0 do
        (let ((s (aref b r)))
          (loop for j from (1+ r) below n
                do (decf s (* (aref a r j) (aref x j))))
          (setf (aref x r) (/ s (aref a r r)))))
      x)))
