

class node:
    def __init__(self,val=0):
        self.val=val
        self.left=None
        self.right=None
res=-1
def find(t,p,q):

    def sup(t):
        global res
        if t is None:
            return False
        lr=sup(t.left)
        rr=sup(t.right)
        if lr==True and rr==True:
            res=t.val
        elif lr==False and rr==False:
            return t.val==p or t.val==q
        else:
            if t.val==p or t.val==q:
                res=t.val
                return True
            else:
                return True
    sup(t)
    return res

n1,n2,n3,n4,n5,n6,n7,n8,n9=node(1),node(2),node(3),node(4),node(5),node(6),node(7),node(8),node(9)
n1.left,n1.right=n2,n3
n2.left,n2.right=n4,n5
n3.left,n3.right=n6,n7
n5.left,n5.right=n8,n9
print(find(n1,8,9))