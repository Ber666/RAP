(define (problem BW-generalization-4)
(:domain blocksworld-4ops)(:objects f i j b k l h g c a e)
(:init 
(handempty)
(ontable f)
(ontable i)
(ontable j)
(ontable b)
(ontable k)
(ontable l)
(ontable h)
(ontable g)
(ontable c)
(ontable a)
(ontable e)
(clear f)
(clear i)
(clear j)
(clear b)
(clear k)
(clear l)
(clear h)
(clear g)
(clear c)
(clear a)
(clear e)
)
(:goal
(and
(on f i)
(on i j)
(on j b)
(on b k)
(on k l)
(on l h)
(on h g)
(on g c)
(on c a)
(on a e)
)))