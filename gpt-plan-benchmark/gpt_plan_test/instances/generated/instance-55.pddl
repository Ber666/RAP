(define (problem BW-generalization-4)
(:domain blocksworld-4ops)(:objects e h b d i f j)
(:init 
(handempty)
(ontable e)
(ontable h)
(ontable b)
(ontable d)
(ontable i)
(ontable f)
(ontable j)
(clear e)
(clear h)
(clear b)
(clear d)
(clear i)
(clear f)
(clear j)
)
(:goal
(and
(on e h)
(on h b)
(on b d)
(on d i)
(on i f)
(on f j)
)))