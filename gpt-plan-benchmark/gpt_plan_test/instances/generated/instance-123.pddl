(define (problem BW-generalization-4)
(:domain blocksworld-4ops)(:objects j f i e b l g d)
(:init 
(handempty)
(ontable j)
(ontable f)
(ontable i)
(ontable e)
(ontable b)
(ontable l)
(ontable g)
(ontable d)
(clear j)
(clear f)
(clear i)
(clear e)
(clear b)
(clear l)
(clear g)
(clear d)
)
(:goal
(and
(on j f)
(on f i)
(on i e)
(on e b)
(on b l)
(on l g)
(on g d)
)))