(define (problem BW-generalization-4)
(:domain blocksworld-4ops)(:objects g c f k h d e j b)
(:init 
(handempty)
(ontable g)
(ontable c)
(ontable f)
(ontable k)
(ontable h)
(ontable d)
(ontable e)
(ontable j)
(ontable b)
(clear g)
(clear c)
(clear f)
(clear k)
(clear h)
(clear d)
(clear e)
(clear j)
(clear b)
)
(:goal
(and
(on g c)
(on c f)
(on f k)
(on k h)
(on h d)
(on d e)
(on e j)
(on j b)
)))