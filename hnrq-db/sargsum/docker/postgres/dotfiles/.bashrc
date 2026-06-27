[ -z "$PS1" ] && return
export PS1="\u@\h:\w\$ "
alias ls='ls -GF'
alias rm='rm -i'
alias cp='cp -i'
alias mv='mv -i'
