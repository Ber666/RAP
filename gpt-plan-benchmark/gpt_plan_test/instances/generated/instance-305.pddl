(define (problem BW-generalization-4)
(:domain blocksworld-4ops)(:objects e a d i g b)
(:init 
(handempty)
(ontable e)
(ontable a)
(ontable d)
(ontable i)
(ontable g)
(ontable b)
(clear e)
(clear a)
(clear d)
(clear i)
(clear g)
(clear b)
)
(:goal
(and
(on e a)
(on a d)
(on d i)
(on i g)
(on g b)
)))