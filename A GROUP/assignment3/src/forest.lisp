;;;; forest.lisp — Model 2: Random Forest regression, built from scratch
;;;; CART variance-reduction trees + bootstrap bagging + feature subsampling
;;;; (mtry = sqrt(p)), matching the Assignment 2 configuration:
;;;; 200 trees, max depth 10, min 2 samples per leaf.

(in-package :cpp4103)

(defstruct tnode
  feature threshold left right value leaf-p)

(defstruct forest trees importance feature-names)

(defun node-sse (y idxs)
  "Sum of squared errors around the mean of Y over IDXS."
  (let* ((n (length idxs))
         (mu (/ (loop for i in idxs sum (aref y i)) n)))
    (values (loop for i in idxs sum (expt (- (aref y i) mu) 2)) mu)))

(defun best-split (x y idxs mtry min-leaf importance)
  "Search MTRY random features for the split minimizing child SSE.
   Returns (values feature threshold gain) or NIL if no valid split."
  (let* ((p (array-dimension x 1))
         (feats (let ((all (loop for j below p collect j)))
                  ;; partial Fisher-Yates to sample mtry features
                  (let ((v (coerce all 'vector)))
                    (loop for i below (min mtry p)
                          do (rotatef (aref v i) (aref v (+ i (rand-int (- p i))))))
                    (loop for i below (min mtry p) collect (aref v i)))))
         (parent-sse (node-sse y idxs))
         (best-gain 0d0) (best-f nil) (best-thr nil))
    (dolist (f feats)
      (let ((sorted (sort (copy-list idxs) #'< :key (lambda (i) (aref x i f)))))
        ;; prefix sums over the sorted order
        (let* ((n (length sorted))
               (ys (map 'vector (lambda (i) (aref y i)) sorted))
               (total (loop for v across ys sum v))
               (total-sq (loop for v across ys sum (* v v)))
               (lsum 0d0) (lsq 0d0))
          (loop for k from 0 below (1- n)
                for i in sorted
                do (let ((v (aref ys k)))
                     (incf lsum v) (incf lsq (* v v)))
                   (let* ((nl (1+ k)) (nr (- n nl))
                          (xi (aref x i f))
                          (xnext (aref x (nth (1+ k) sorted) f)))
                     (when (and (>= nl min-leaf) (>= nr min-leaf)
                                (> (- xnext xi) 1d-12))
                       (let* ((rsum (- total lsum))
                              (rsq (- total-sq lsq))
                              (sse-l (- lsq (/ (* lsum lsum) nl)))
                              (sse-r (- rsq (/ (* rsum rsum) nr)))
                              (gain (- parent-sse (+ sse-l sse-r))))
                         (when (> gain best-gain)
                           (setf best-gain gain
                                 best-f f
                                 best-thr (/ (+ xi xnext) 2d0))))))))))
    (when best-f
      (when importance (incf (aref importance best-f) best-gain))
      (values best-f best-thr best-gain))))

(defun grow-tree (x y idxs depth max-depth min-leaf mtry importance)
  (multiple-value-bind (sse mu) (node-sse y idxs)
    (if (or (>= depth max-depth)
            (< (length idxs) (* 2 min-leaf))
            (< sse 1d-9))
        (make-tnode :leaf-p t :value mu)
        (multiple-value-bind (f thr) (best-split x y idxs mtry min-leaf importance)
          (if (null f)
              (make-tnode :leaf-p t :value mu)
              (let ((left '()) (right '()))
                (dolist (i idxs)
                  (if (<= (aref x i f) thr) (push i left) (push i right)))
                (make-tnode :feature f :threshold thr
                            :left (grow-tree x y left (1+ depth) max-depth min-leaf mtry importance)
                            :right (grow-tree x y right (1+ depth) max-depth min-leaf mtry importance))))))))

(defun tree-predict (node x i)
  (if (tnode-leaf-p node)
      (tnode-value node)
      (if (<= (aref x i (tnode-feature node)) (tnode-threshold node))
          (tree-predict (tnode-left node) x i)
          (tree-predict (tnode-right node) x i))))

(defun forest-fit (x y &key (n-trees 200) (max-depth 10) (min-leaf 2) feature-names)
  (let* ((n (array-dimension x 0))
         (p (array-dimension x 1))
         (mtry (max 1 (round (sqrt p))))
         (importance (make-array p :element-type 'double-float :initial-element 0d0))
         (trees
           (loop repeat n-trees
                 collect (let ((boot (loop repeat n collect (rand-int n))))
                           (grow-tree x y boot 0 max-depth min-leaf mtry importance)))))
    ;; normalize importance to sum 1
    (let ((total (loop for v across importance sum v)))
      (when (plusp total)
        (loop for j below p do (setf (aref importance j) (/ (aref importance j) total)))))
    (make-forest :trees trees :importance importance :feature-names feature-names)))

(defun forest-predict (model x)
  (let* ((n (array-dimension x 0))
         (trees (forest-trees model))
         (k (length trees))
         (out (make-array n :element-type 'double-float :initial-element 0d0)))
    (loop for i below n do
      (setf (aref out i)
            (/ (loop for tr in trees sum (tree-predict tr x i)) k)))
    out))
