(define (problem BW-generalization-4)
(:domain blocksworld-4ops)(:objects f k l e g h d b j c i a)
(:init 
(handempty)
(ontable f)
(ontable k)
(ontable l)
(ontable e)
(ontable g)
(ontable h)
(ontable d)
(ontable b)
(ontable j)
(ontable c)
(ontable i)
(ontable a)
(clear f)
(clear k)
(clear l)
(clear e)
(clear g)
(clear h)
(clear d)
(clear b)
(clear j)
(clear c)
(clear i)
(clear a)
)
(:goal
(and
(on f k)
(on k l)
(on l e)
(on e g)
(on g h)
(on h d)
(on d b)
(on b j)
(on j c)
(on c i)
(on i a)
)))