(define (problem BW-generalization-4)
(:domain blocksworld-4ops)(:objects i a j d b c h g e f)
(:init 
(handempty)
(ontable i)
(ontable a)
(ontable j)
(ontable d)
(ontable b)
(ontable c)
(ontable h)
(ontable g)
(ontable e)
(ontable f)
(clear i)
(clear a)
(clear j)
(clear d)
(clear b)
(clear c)
(clear h)
(clear g)
(clear e)
(clear f)
)
(:goal
(and
(on i a)
(on a j)
(on j d)
(on d b)
(on b c)
(on c h)
(on h g)
(on g e)
(on e f)
)))