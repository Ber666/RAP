(define (problem BW-generalization-4)
(:domain blocksworld-4ops)(:objects c d g b h)
(:init 
(handempty)
(ontable c)
(ontable d)
(ontable g)
(ontable b)
(ontable h)
(clear c)
(clear d)
(clear g)
(clear b)
(clear h)
)
(:goal
(and
(on c d)
(on d g)
(on g b)
(on b h)
)))