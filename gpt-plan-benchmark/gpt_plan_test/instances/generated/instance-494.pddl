(define (problem BW-generalization-4)
(:domain blocksworld-4ops)(:objects a b j i l e g c)
(:init 
(handempty)
(ontable a)
(ontable b)
(ontable j)
(ontable i)
(ontable l)
(ontable e)
(ontable g)
(ontable c)
(clear a)
(clear b)
(clear j)
(clear i)
(clear l)
(clear e)
(clear g)
(clear c)
)
(:goal
(and
(on a b)
(on b j)
(on j i)
(on i l)
(on l e)
(on e g)
(on g c)
)))