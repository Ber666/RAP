(define (problem BW-generalization-4)
(:domain blocksworld-4ops)(:objects e f b a h j c k i l)
(:init 
(handempty)
(ontable e)
(ontable f)
(ontable b)
(ontable a)
(ontable h)
(ontable j)
(ontable c)
(ontable k)
(ontable i)
(ontable l)
(clear e)
(clear f)
(clear b)
(clear a)
(clear h)
(clear j)
(clear c)
(clear k)
(clear i)
(clear l)
)
(:goal
(and
(on e f)
(on f b)
(on b a)
(on a h)
(on h j)
(on j c)
(on c k)
(on k i)
(on i l)
)))