#ifndef CAKEML_AST_H
#define CAKEML_AST_H

#include <string>
#include <vector>
#include <memory>

namespace cakeml {

class ExprNode {
public:
  enum ExprNodeKind {
    Kind_Num,
    Kind_Var,
    Kind_BinOp,
    Kind_Let,
    Kind_Lambda,
    Kind_Seq,
    Kind_Call,
    Kind_If,
    Kind_Print,
  };

  ExprNode(ExprNodeKind kind)
      : kind(kind) {}
  virtual ~ExprNode() = default;

  ExprNodeKind getKind() const { return kind; }
  virtual std::string dump() = 0;

private:
  const ExprNodeKind kind;
};

class NumberExprNode : public ExprNode {
  int val;

public:
  NumberExprNode(int val)
      : ExprNode(Kind_Num), val(val) {}

  int getValue() { return val; }
  std::string dump() override {
    return std::to_string(val);
  }
};

class VarExprNode : public ExprNode {
  std::string name;

public:
  VarExprNode(std::string name)
      : ExprNode(Kind_Var), name(name) {}

  std::string getName() { return name; }
  std::string dump() override {
    return name;
  }
};

class BinopExprNode : public ExprNode {
  char op;
  std::unique_ptr<ExprNode> l, r;

public:
  BinopExprNode(char op, std::unique_ptr<ExprNode> l, std::unique_ptr<ExprNode> r)
      : ExprNode(Kind_BinOp), op(op), l(std::move(l)), r(std::move(r)) {}

  char getOp() { return op; }
  ExprNode* getL() { return l.get(); }
  ExprNode* getR() { return r.get(); }
  std::string dump() override {
    return l->dump() + std::string(1, op) + r->dump();
  }
};

class LetExprNode : public ExprNode {
  std::string var;
  std::unique_ptr<ExprNode> decl, body;

public:
  LetExprNode(std::string var, std::unique_ptr<ExprNode> decl, std::unique_ptr<ExprNode> body)
      : ExprNode(Kind_Let), var(var), decl(std::move(decl)), body(std::move(body)) {}

  std::string getVar() { return var; }
  ExprNode* getDecl() { return decl.get(); }
  ExprNode* getBody() { return body.get(); }
  std::string dump() override {
    return "let(" + var + "=" + decl->dump()  + ") {" + body->dump() + "}";
  }
};

class LambdaExprNode : public ExprNode {
  std::string fn;
  std::vector<std::string> args;
  std::unique_ptr<ExprNode> body;

public:
  LambdaExprNode(std::string fn, std::vector<std::string> args, std::unique_ptr<ExprNode> body)
      : ExprNode(Kind_Lambda), fn(fn), args(args), body(std::move(body)) {}

  std::string getFn() { return fn; }
  std::vector<std::string>& getArgs() { return args; }
  ExprNode* getBody() { return body.get(); }
  std::string dump() override {
    std::string argList = "";
    for (size_t i = 0; i < args.size(); i ++) {
      if (i != 0) argList += ", ";
      auto& arg = args[i];
      argList += arg;
    }
    return "lambda " + fn + "(" + argList  + ") {" + body->dump() + "}";
  }
};

class SeqExprNode : public ExprNode {
  std::string var;
  std::unique_ptr<ExprNode> hd, tl;

public:
  SeqExprNode(std::unique_ptr<ExprNode> hd, std::unique_ptr<ExprNode> tl)
      : ExprNode(Kind_Seq), hd(std::move(hd)), tl(std::move(tl)) {}

  ExprNode* getHd() { return hd.get(); }
  ExprNode* getTl() { return tl.get(); }
  std::string dump() override {
    return hd->dump()  + ";\n" + tl->dump();
  }
};

class CallExprNode : public ExprNode {
  std::unique_ptr<ExprNode> fn;
  std::vector<std::unique_ptr<ExprNode>> args;

public:
  CallExprNode(std::unique_ptr<ExprNode> fn, std::vector<std::unique_ptr<ExprNode>> args)
      : ExprNode(Kind_Call), fn(std::move(fn)), args(std::move(args)) {}

  ExprNode* getFn() { return fn.get(); }
  size_t getArgCount() { return args.size(); }
  ExprNode* getArg(size_t i) { return args[i].get(); }
  std::string dump() override {
    std::string res = "call " + fn->dump() + "(";
    for (size_t i = 0; i < args.size(); i ++) {
      if (i != 0) res += ", ";
      res += args[i]->dump();
    }
    return res + ")";
  }
};

class IfExprNode : public ExprNode {
  std::string var;
  std::unique_ptr<ExprNode> cond, then, els;

public:
  IfExprNode(std::unique_ptr<ExprNode> cond, std::unique_ptr<ExprNode> then, std::unique_ptr<ExprNode> els)
      : ExprNode(Kind_If), cond(std::move(cond)), then(std::move(then)), els(std::move(els)) {}

  ExprNode* getCond() { return cond.get(); }
  ExprNode* getThen() { return then.get(); }
  ExprNode* getEls() { return els.get(); }
  std::string dump() override {
    return "if (" + cond->dump() + ")\n  {" + then->dump()  + "}\n  {" + els->dump() + "}";
  }
};

class PrintExprNode : public ExprNode {
  std::unique_ptr<ExprNode> expr;

public:
  PrintExprNode(std::unique_ptr<ExprNode> expr)
      : ExprNode(Kind_Print), expr(std::move(expr)) {}

  ExprNode* getExpr() { return expr.get(); }
  std::string dump() override {
    return "print(" + expr->dump() + ")";
  }
};
}

#endif // CAKEML_AST_H