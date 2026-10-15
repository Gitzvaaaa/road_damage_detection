import re
from lxml import etree

NS='http://www.w3.org/1998/Math/MathML'
def el(tag,*children,text=None,**attrs):
    x=etree.Element('{'+NS+'}'+tag,**attrs)
    if text is not None: x.text=text
    for c in children: x.append(c)
    return x

symbols={
 'theta':'θ','mu':'μ','sigma':'σ','gamma':'γ','beta':'β','epsilon':'ϵ','eta':'η','lambda':'λ','delta':'δ','Delta':'Δ',
 'times':'×','in':'∈','ne':'≠','geq':'≥','leq':'≤','odot':'⊙','nabla':'∇','partial':'∂','infty':'∞',
 'sum':'∑','lfloor':'⌊','rfloor':'⌋','|':'‖','{':'{','}':'}','Longleftrightarrow':'⇔','sim':'∼',
}
names={'max','min','log','exp','tanh','arg'}

class Parser:
    def __init__(self,src): self.t=re.findall(r'\\[a-zA-Z]+|\\.|\d+(?:\.\d+)?|[^\s]',src); self.i=0
    def pop(self):
        t=self.t[self.i]; self.i+=1; return t
    def peek(self): return self.t[self.i] if self.i<len(self.t) else None
    def group_text(self):
        if self.pop()!='{': raise ValueError('Expected group')
        depth=1; acc=[]
        while depth:
            t=self.pop()
            if t=='{': depth+=1
            if t=='}': depth-=1
            if depth: acc.append(t)
        return ''.join(acc)
    def arg(self):
        if self.peek()=='{':
            self.pop(); x=self.seq('}'); self.pop(); return x
        return self.atom()
    def seq(self,stop=None):
        out=[]
        while self.peek() is not None and self.peek()!=stop:
            node=self.atom()
            if node is None: continue
            sub=sup=None
            while self.peek() in ['_','^']:
                typ=self.pop(); a=self.arg()
                if typ=='_': sub=a
                else: sup=a
            if sub is not None and sup is not None: node=el('msubsup',node,sub,sup)
            elif sub is not None: node=el('msub',node,sub)
            elif sup is not None: node=el('msup',node,sup)
            out.append(node)
        return el('mrow',*out)
    def atom(self):
        t=self.pop()
        if t=='{':
            x=self.seq('}'); self.pop(); return x
        if t.startswith('\\'):
            cmd=t[1:]
            if cmd in ['left','right']: return None
            if cmd in [',',';','quad','qquad']:
                return el('mtext',text='\u2003' if cmd in ['quad','qquad'] else '\u2009')
            if cmd=='frac': return el('mfrac',self.arg(),self.arg())
            if cmd=='sqrt': return el('msqrt',self.arg())
            if cmd in ['widehat','hat','widetilde','tilde']:
                return el('mover',self.arg(),el('mo',text='~' if 'tilde' in cmd else '^'),accent='true')
            if cmd in ['mathrm','mathbf','mathbb','mathcal']:
                variants={'mathrm':'normal','mathbf':'bold','mathbb':'double-struck','mathcal':'script'}
                arg=self.arg()
                for token in arg.iter():
                    if etree.QName(token).localname in ['mi','mn']:
                        token.set('mathvariant',variants[cmd])
                return arg
            if cmd in ['text','operatorname']:
                # Preserve spaces in human-readable text using the token locations indirectly.
                s=self.group_text()
                return el('mtext',text=s)
            if cmd=='begin':
                env=self.group_text(); start=self.i; depth=0
                while not (self.t[self.i]=='\\end' and depth==0):
                    if self.t[self.i]=='\\begin': depth+=1
                    if self.t[self.i]=='\\end': depth-=1
                    self.i+=1
                content=self.t[start:self.i]; self.pop(); ending=self.group_text()
                assert ending==env
                rows=[]; cells=[]; acc=[]; level=0
                for tok in content+['\\\\']:
                    if tok=='{': level+=1
                    if tok=='}': level-=1
                    if level==0 and tok in ['&','\\\\']:
                        ps=Parser(''); ps.t=acc; cells.append(el('mtd',ps.seq())); acc=[]
                        if tok=='\\\\': rows.append(el('mtr',*cells)); cells=[]
                    else: acc.append(tok)
                table=el('mtable',*rows,columnalign='left' if env=='cases' else 'center')
                return el('mfenced',table,open='{' if env=='cases' else '[',close='' if env=='cases' else ']')
            if cmd in names: return el('mi',text=cmd,mathvariant='normal')
            if cmd in symbols: return el('mo' if cmd in ['sum','times','in','ne','geq','leq','odot','nabla','partial','Longleftrightarrow','lfloor','rfloor','|','{','}'] else 'mi',text=symbols[cmd])
            raise ValueError('Unsupported command '+cmd)
        if re.fullmatch(r'\d+(?:\.\d+)?',t): return el('mn',text=t)
        if t.isalpha(): return el('mi',text=t)
        if t in ['_','^','}']: raise ValueError('Unexpected '+t)
        return el('mo',text=t)

def convert(src,display=False):
    # Token placeholders preserve spaces inside text groups while parsing formulas.
    texts=[]
    def save(m):
        texts.append(m.group(1)); return r'\text{TEXTPLACEHOLDER'+str(len(texts)-1)+'}'
    src=re.sub(r'\\text\{([^{}]*)\}',save,src)
    tree=el('math',Parser(src).seq(),display='block' if display else 'inline')
    for x in tree.iter('{'+NS+'}mtext'):
        if x.text and x.text.startswith('TEXTPLACEHOLDER'): x.text=texts[int(x.text[len('TEXTPLACEHOLDER'):])]
    return tree
