(define (problem BW-generalization-4)
(:domain blocksworld-4ops)(:objects a k c j l g i)
(:init 
(handempty)
(ontable a)
(ontable k)
(ontable c)
(ontable j)
(ontable l)
(ontable g)
(ontable i)
(clear a)
(clear k)
(clear c)
(clear j)
(clear l)
(clear g)
(clear i)
)
(:goal
(and
(on a k)
(on k c)
(on c j)
(on j l)
(on l g)
(on g i)
)))