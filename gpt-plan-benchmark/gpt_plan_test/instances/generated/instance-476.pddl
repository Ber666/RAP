(define (problem BW-generalization-4)
(:domain blocksworld-4ops)(:objects j a e k i h c f g)
(:init 
(handempty)
(ontable j)
(ontable a)
(ontable e)
(ontable k)
(ontable i)
(ontable h)
(ontable c)
(ontable f)
(ontable g)
(clear j)
(clear a)
(clear e)
(clear k)
(clear i)
(clear h)
(clear c)
(clear f)
(clear g)
)
(:goal
(and
(on j a)
(on a e)
(on e k)
(on k i)
(on i h)
(on h c)
(on c f)
(on f g)
)))