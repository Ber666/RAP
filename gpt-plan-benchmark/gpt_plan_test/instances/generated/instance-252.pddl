(define (problem BW-generalization-4)
(:domain blocksworld-4ops)(:objects b a k j i l h c d g e f)
(:init 
(handempty)
(ontable b)
(ontable a)
(ontable k)
(ontable j)
(ontable i)
(ontable l)
(ontable h)
(ontable c)
(ontable d)
(ontable g)
(ontable e)
(ontable f)
(clear b)
(clear a)
(clear k)
(clear j)
(clear i)
(clear l)
(clear h)
(clear c)
(clear d)
(clear g)
(clear e)
(clear f)
)
(:goal
(and
(on b a)
(on a k)
(on k j)
(on j i)
(on i l)
(on l h)
(on h c)
(on c d)
(on d g)
(on g e)
(on e f)
)))