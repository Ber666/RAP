(define (problem BW-generalization-4)
(:domain blocksworld-4ops)(:objects c e k d l g j)
(:init 
(handempty)
(ontable c)
(ontable e)
(ontable k)
(ontable d)
(ontable l)
(ontable g)
(ontable j)
(clear c)
(clear e)
(clear k)
(clear d)
(clear l)
(clear g)
(clear j)
)
(:goal
(and
(on c e)
(on e k)
(on k d)
(on d l)
(on l g)
(on g j)
)))