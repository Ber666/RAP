(define (problem BW-generalization-4)
(:domain blocksworld-4ops)(:objects b g k j l e d a i c)
(:init 
(handempty)
(ontable b)
(ontable g)
(ontable k)
(ontable j)
(ontable l)
(ontable e)
(ontable d)
(ontable a)
(ontable i)
(ontable c)
(clear b)
(clear g)
(clear k)
(clear j)
(clear l)
(clear e)
(clear d)
(clear a)
(clear i)
(clear c)
)
(:goal
(and
(on b g)
(on g k)
(on k j)
(on j l)
(on l e)
(on e d)
(on d a)
(on a i)
(on i c)
)))