(define (problem BW-generalization-4)
(:domain blocksworld-4ops)(:objects i d c e g h j k a f)
(:init 
(handempty)
(ontable i)
(ontable d)
(ontable c)
(ontable e)
(ontable g)
(ontable h)
(ontable j)
(ontable k)
(ontable a)
(ontable f)
(clear i)
(clear d)
(clear c)
(clear e)
(clear g)
(clear h)
(clear j)
(clear k)
(clear a)
(clear f)
)
(:goal
(and
(on i d)
(on d c)
(on c e)
(on e g)
(on g h)
(on h j)
(on j k)
(on k a)
(on a f)
)))