(define (problem BW-generalization-4)
(:domain blocksworld-4ops)(:objects b e c f j g d h i)
(:init 
(handempty)
(ontable b)
(ontable e)
(ontable c)
(ontable f)
(ontable j)
(ontable g)
(ontable d)
(ontable h)
(ontable i)
(clear b)
(clear e)
(clear c)
(clear f)
(clear j)
(clear g)
(clear d)
(clear h)
(clear i)
)
(:goal
(and
(on b e)
(on e c)
(on c f)
(on f j)
(on j g)
(on g d)
(on d h)
(on h i)
)))