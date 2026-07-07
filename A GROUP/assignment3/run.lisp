;;;; run.lisp — loader/entry script
;;;; Usage:  sbcl --non-interactive --load run.lisp
;;;; (run from the assignment3/ directory, or from anywhere — paths are
;;;; resolved relative to this file)

(let ((base (make-pathname :name nil :type nil :defaults *load-truename*)))
  (load (merge-pathnames "src/utils.lisp" base))
  (setf (symbol-value (intern "*BASE-DIR*" :cpp4103)) base)
  (load (merge-pathnames "src/dataset.lisp" base))
  (load (merge-pathnames "src/linreg.lisp" base))
  (load (merge-pathnames "src/forest.lisp" base))
  (load (merge-pathnames "src/lstm.lisp" base))
  (load (merge-pathnames "src/main.lisp" base)))

(cpp4103:run)
