(define (problem BW-generalization-4)
(:domain blocksworld-4ops)(:objects k e i f d j g b)
(:init 
(handempty)
(ontable k)
(ontable e)
(ontable i)
(ontable f)
(ontable d)
(ontable j)
(ontable g)
(ontable b)
(clear k)
(clear e)
(clear i)
(clear f)
(clear d)
(clear j)
(clear g)
(clear b)
)
(:goal
(and
(on k e)
(on e i)
(on i f)
(on f d)
(on d j)
(on j g)
(on g b)
)))