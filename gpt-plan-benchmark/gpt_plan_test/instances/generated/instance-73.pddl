(define (problem BW-generalization-4)
(:domain blocksworld-4ops)(:objects l h e j b d a g f i c)
(:init 
(handempty)
(ontable l)
(ontable h)
(ontable e)
(ontable j)
(ontable b)
(ontable d)
(ontable a)
(ontable g)
(ontable f)
(ontable i)
(ontable c)
(clear l)
(clear h)
(clear e)
(clear j)
(clear b)
(clear d)
(clear a)
(clear g)
(clear f)
(clear i)
(clear c)
)
(:goal
(and
(on l h)
(on h e)
(on e j)
(on j b)
(on b d)
(on d a)
(on a g)
(on g f)
(on f i)
(on i c)
)))