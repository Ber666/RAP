(define (problem BW-generalization-4)
(:domain blocksworld-4ops)(:objects c e h i f a)
(:init 
(handempty)
(ontable c)
(ontable e)
(ontable h)
(ontable i)
(ontable f)
(ontable a)
(clear c)
(clear e)
(clear h)
(clear i)
(clear f)
(clear a)
)
(:goal
(and
(on c e)
(on e h)
(on h i)
(on i f)
(on f a)
)))