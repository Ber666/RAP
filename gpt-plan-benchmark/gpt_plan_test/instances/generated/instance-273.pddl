(define (problem BW-generalization-4)
(:domain blocksworld-4ops)(:objects c e f h a i g)
(:init 
(handempty)
(ontable c)
(ontable e)
(ontable f)
(ontable h)
(ontable a)
(ontable i)
(ontable g)
(clear c)
(clear e)
(clear f)
(clear h)
(clear a)
(clear i)
(clear g)
)
(:goal
(and
(on c e)
(on e f)
(on f h)
(on h a)
(on a i)
(on i g)
)))