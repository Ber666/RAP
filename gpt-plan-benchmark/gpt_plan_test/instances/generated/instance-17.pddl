(define (problem BW-generalization-4)
(:domain blocksworld-4ops)(:objects b c k l h a j e d f)
(:init 
(handempty)
(ontable b)
(ontable c)
(ontable k)
(ontable l)
(ontable h)
(ontable a)
(ontable j)
(ontable e)
(ontable d)
(ontable f)
(clear b)
(clear c)
(clear k)
(clear l)
(clear h)
(clear a)
(clear j)
(clear e)
(clear d)
(clear f)
)
(:goal
(and
(on b c)
(on c k)
(on k l)
(on l h)
(on h a)
(on a j)
(on j e)
(on e d)
(on d f)
)))