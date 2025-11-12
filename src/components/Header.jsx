import chefClaudeLogo from "/src/images/chef-claude-icon.png"

export default function Header(){
    return(
        <header>
            <img src={chefClaudeLogo} alt="chef-claude img" />
            <h1>Chef Claude</h1>
        </header>
    )
}