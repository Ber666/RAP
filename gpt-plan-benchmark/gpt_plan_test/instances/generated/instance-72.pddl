(define (problem BW-generalization-4)
(:domain blocksworld-4ops)(:objects a e f d h c)
(:init 
(handempty)
(ontable a)
(ontable e)
(ontable f)
(ontable d)
(ontable h)
(ontable c)
(clear a)
(clear e)
(clear f)
(clear d)
(clear h)
(clear c)
)
(:goal
(and
(on a e)
(on e f)
(on f d)
(on d h)
(on h c)
)))