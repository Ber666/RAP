(define (problem BW-generalization-4)
(:domain blocksworld-4ops)(:objects c g j a e k i f)
(:init 
(handempty)
(ontable c)
(ontable g)
(ontable j)
(ontable a)
(ontable e)
(ontable k)
(ontable i)
(ontable f)
(clear c)
(clear g)
(clear j)
(clear a)
(clear e)
(clear k)
(clear i)
(clear f)
)
(:goal
(and
(on c g)
(on g j)
(on j a)
(on a e)
(on e k)
(on k i)
(on i f)
)))