(define (problem BW-generalization-4)
(:domain blocksworld-4ops)(:objects k g c i b)
(:init 
(handempty)
(ontable k)
(ontable g)
(ontable c)
(ontable i)
(ontable b)
(clear k)
(clear g)
(clear c)
(clear i)
(clear b)
)
(:goal
(and
(on k g)
(on g c)
(on c i)
(on i b)
)))