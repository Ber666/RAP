(define (problem BW-generalization-4)
(:domain blocksworld-4ops)(:objects f i d b k g a h j c l e)
(:init 
(handempty)
(ontable f)
(ontable i)
(ontable d)
(ontable b)
(ontable k)
(ontable g)
(ontable a)
(ontable h)
(ontable j)
(ontable c)
(ontable l)
(ontable e)
(clear f)
(clear i)
(clear d)
(clear b)
(clear k)
(clear g)
(clear a)
(clear h)
(clear j)
(clear c)
(clear l)
(clear e)
)
(:goal
(and
(on f i)
(on i d)
(on d b)
(on b k)
(on k g)
(on g a)
(on a h)
(on h j)
(on j c)
(on c l)
(on l e)
)))