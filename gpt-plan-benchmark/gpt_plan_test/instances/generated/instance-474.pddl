(define (problem BW-generalization-4)
(:domain blocksworld-4ops)(:objects d l c e f a k h)
(:init 
(handempty)
(ontable d)
(ontable l)
(ontable c)
(ontable e)
(ontable f)
(ontable a)
(ontable k)
(ontable h)
(clear d)
(clear l)
(clear c)
(clear e)
(clear f)
(clear a)
(clear k)
(clear h)
)
(:goal
(and
(on d l)
(on l c)
(on c e)
(on e f)
(on f a)
(on a k)
(on k h)
)))