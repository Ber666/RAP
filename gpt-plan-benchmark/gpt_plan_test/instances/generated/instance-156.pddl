(define (problem BW-generalization-4)
(:domain blocksworld-4ops)(:objects h k b i a g c)
(:init 
(handempty)
(ontable h)
(ontable k)
(ontable b)
(ontable i)
(ontable a)
(ontable g)
(ontable c)
(clear h)
(clear k)
(clear b)
(clear i)
(clear a)
(clear g)
(clear c)
)
(:goal
(and
(on h k)
(on k b)
(on b i)
(on i a)
(on a g)
(on g c)
)))