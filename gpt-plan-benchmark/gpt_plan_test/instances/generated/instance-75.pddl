(define (problem BW-generalization-4)
(:domain blocksworld-4ops)(:objects e l i a g d)
(:init 
(handempty)
(ontable e)
(ontable l)
(ontable i)
(ontable a)
(ontable g)
(ontable d)
(clear e)
(clear l)
(clear i)
(clear a)
(clear g)
(clear d)
)
(:goal
(and
(on e l)
(on l i)
(on i a)
(on a g)
(on g d)
)))