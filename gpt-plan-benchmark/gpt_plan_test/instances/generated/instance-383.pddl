(define (problem BW-generalization-4)
(:domain blocksworld-4ops)(:objects d g a k e f)
(:init 
(handempty)
(ontable d)
(ontable g)
(ontable a)
(ontable k)
(ontable e)
(ontable f)
(clear d)
(clear g)
(clear a)
(clear k)
(clear e)
(clear f)
)
(:goal
(and
(on d g)
(on g a)
(on a k)
(on k e)
(on e f)
)))