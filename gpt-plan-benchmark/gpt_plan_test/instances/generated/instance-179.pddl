(define (problem BW-generalization-4)
(:domain blocksworld-4ops)(:objects a j d g f l e)
(:init 
(handempty)
(ontable a)
(ontable j)
(ontable d)
(ontable g)
(ontable f)
(ontable l)
(ontable e)
(clear a)
(clear j)
(clear d)
(clear g)
(clear f)
(clear l)
(clear e)
)
(:goal
(and
(on a j)
(on j d)
(on d g)
(on g f)
(on f l)
(on l e)
)))