(define (problem BW-generalization-4)
(:domain blocksworld-4ops)(:objects a b h d f)
(:init 
(handempty)
(ontable a)
(ontable b)
(ontable h)
(ontable d)
(ontable f)
(clear a)
(clear b)
(clear h)
(clear d)
(clear f)
)
(:goal
(and
(on a b)
(on b h)
(on h d)
(on d f)
)))