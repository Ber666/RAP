(define (problem BW-generalization-4)
(:domain blocksworld-4ops)(:objects b a e h g i d c f l)
(:init 
(handempty)
(ontable b)
(ontable a)
(ontable e)
(ontable h)
(ontable g)
(ontable i)
(ontable d)
(ontable c)
(ontable f)
(ontable l)
(clear b)
(clear a)
(clear e)
(clear h)
(clear g)
(clear i)
(clear d)
(clear c)
(clear f)
(clear l)
)
(:goal
(and
(on b a)
(on a e)
(on e h)
(on h g)
(on g i)
(on i d)
(on d c)
(on c f)
(on f l)
)))