(define (problem BW-generalization-4)
(:domain blocksworld-4ops)(:objects k e a b c j d)
(:init 
(handempty)
(ontable k)
(ontable e)
(ontable a)
(ontable b)
(ontable c)
(ontable j)
(ontable d)
(clear k)
(clear e)
(clear a)
(clear b)
(clear c)
(clear j)
(clear d)
)
(:goal
(and
(on k e)
(on e a)
(on a b)
(on b c)
(on c j)
(on j d)
)))