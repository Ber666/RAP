(define (problem BW-generalization-4)
(:domain blocksworld-4ops)(:objects j i a l b k e c)
(:init 
(handempty)
(ontable j)
(ontable i)
(ontable a)
(ontable l)
(ontable b)
(ontable k)
(ontable e)
(ontable c)
(clear j)
(clear i)
(clear a)
(clear l)
(clear b)
(clear k)
(clear e)
(clear c)
)
(:goal
(and
(on j i)
(on i a)
(on a l)
(on l b)
(on b k)
(on k e)
(on e c)
)))