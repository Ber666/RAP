(define (problem BW-generalization-4)
(:domain blocksworld-4ops)(:objects f b d k j l h e i)
(:init 
(handempty)
(ontable f)
(ontable b)
(ontable d)
(ontable k)
(ontable j)
(ontable l)
(ontable h)
(ontable e)
(ontable i)
(clear f)
(clear b)
(clear d)
(clear k)
(clear j)
(clear l)
(clear h)
(clear e)
(clear i)
)
(:goal
(and
(on f b)
(on b d)
(on d k)
(on k j)
(on j l)
(on l h)
(on h e)
(on e i)
)))