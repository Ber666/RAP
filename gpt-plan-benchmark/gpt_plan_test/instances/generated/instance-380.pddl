(define (problem BW-generalization-4)
(:domain blocksworld-4ops)(:objects c l j e d g a f k i b h)
(:init 
(handempty)
(ontable c)
(ontable l)
(ontable j)
(ontable e)
(ontable d)
(ontable g)
(ontable a)
(ontable f)
(ontable k)
(ontable i)
(ontable b)
(ontable h)
(clear c)
(clear l)
(clear j)
(clear e)
(clear d)
(clear g)
(clear a)
(clear f)
(clear k)
(clear i)
(clear b)
(clear h)
)
(:goal
(and
(on c l)
(on l j)
(on j e)
(on e d)
(on d g)
(on g a)
(on a f)
(on f k)
(on k i)
(on i b)
(on b h)
)))