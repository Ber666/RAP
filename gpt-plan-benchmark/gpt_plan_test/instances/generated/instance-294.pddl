(define (problem BW-generalization-4)
(:domain blocksworld-4ops)(:objects d j g f l e h i c k)
(:init 
(handempty)
(ontable d)
(ontable j)
(ontable g)
(ontable f)
(ontable l)
(ontable e)
(ontable h)
(ontable i)
(ontable c)
(ontable k)
(clear d)
(clear j)
(clear g)
(clear f)
(clear l)
(clear e)
(clear h)
(clear i)
(clear c)
(clear k)
)
(:goal
(and
(on d j)
(on j g)
(on g f)
(on f l)
(on l e)
(on e h)
(on h i)
(on i c)
(on c k)
)))