(define (problem BW-generalization-4)
(:domain blocksworld-4ops)(:objects h c a e j b g i f k l)
(:init 
(handempty)
(ontable h)
(ontable c)
(ontable a)
(ontable e)
(ontable j)
(ontable b)
(ontable g)
(ontable i)
(ontable f)
(ontable k)
(ontable l)
(clear h)
(clear c)
(clear a)
(clear e)
(clear j)
(clear b)
(clear g)
(clear i)
(clear f)
(clear k)
(clear l)
)
(:goal
(and
(on h c)
(on c a)
(on a e)
(on e j)
(on j b)
(on b g)
(on g i)
(on i f)
(on f k)
(on k l)
)))