;;;; linreg.lisp — Model 1: Linear Regression with Ridge (L2) regularization
;;;; Solves (X'X + lambda*I) w = X'y by Gaussian elimination.
;;;; The bias term is appended as a constant column and left unpenalized.

(in-package :cpp4103)

(defstruct linreg weights)  ; length p+1, last entry is the bias

(defun linreg-fit (x y &key (l2 0.1d0))
  (let* ((n (array-dimension x 0))
         (p (array-dimension x 1))
         (q (1+ p))
         (a (make-array (list q q) :element-type 'double-float :initial-element 0d0))
         (b (make-array q :element-type 'double-float :initial-element 0d0)))
    (flet ((xcell (i j) (if (< j p) (aref x i j) 1d0)))
      ;; A = X'X (+ ridge on non-bias diagonal), b = X'y
      (loop for j below q do
        (loop for k from j below q do
          (let ((s 0d0))
            (loop for i below n do (incf s (* (xcell i j) (xcell i k))))
            (setf (aref a j k) s
                  (aref a k j) s)))
        (when (< j p) (incf (aref a j j) l2))
        (let ((s 0d0))
          (loop for i below n do (incf s (* (xcell i j) (aref y i))))
          (setf (aref b j) s))))
    (make-linreg :weights (solve-linear-system a b))))

(defun linreg-predict-row (model x i)
  (let* ((w (linreg-weights model))
         (p (1- (length w)))
         (s (aref w p)))   ; bias
    (loop for j below p do (incf s (* (aref w j) (aref x i j))))
    s))

(defun linreg-predict (model x)
  (let* ((n (array-dimension x 0))
         (out (make-array n :element-type 'double-float)))
    (loop for i below n do (setf (aref out i) (linreg-predict-row model x i)))
    out))
