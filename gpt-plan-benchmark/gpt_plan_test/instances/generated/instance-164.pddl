(define (problem BW-generalization-4)
(:domain blocksworld-4ops)(:objects g h k e f d i b c l j a)
(:init 
(handempty)
(ontable g)
(ontable h)
(ontable k)
(ontable e)
(ontable f)
(ontable d)
(ontable i)
(ontable b)
(ontable c)
(ontable l)
(ontable j)
(ontable a)
(clear g)
(clear h)
(clear k)
(clear e)
(clear f)
(clear d)
(clear i)
(clear b)
(clear c)
(clear l)
(clear j)
(clear a)
)
(:goal
(and
(on g h)
(on h k)
(on k e)
(on e f)
(on f d)
(on d i)
(on i b)
(on b c)
(on c l)
(on l j)
(on j a)
)))