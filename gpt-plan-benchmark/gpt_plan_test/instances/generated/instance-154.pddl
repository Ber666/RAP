(define (problem BW-generalization-4)
(:domain blocksworld-4ops)(:objects k c b a g e)
(:init 
(handempty)
(ontable k)
(ontable c)
(ontable b)
(ontable a)
(ontable g)
(ontable e)
(clear k)
(clear c)
(clear b)
(clear a)
(clear g)
(clear e)
)
(:goal
(and
(on k c)
(on c b)
(on b a)
(on a g)
(on g e)
)))