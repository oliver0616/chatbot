import { NgModule } from '@angular/core';
import { Routes, RouterModule } from '@angular/router';


import {LayoutComponent} from './dashboard/layout/layout.component'
import { LoginComponent } from './login/index';

const routes: Routes = [
	{ path: '', redirectTo: 'login', pathMatch:'full'},
	{
		path: 'login',
    component: LoginComponent,
	},
	//{ path: '', redirectTo: HomeComponent, pathMatch: 'full'},
  //{ path: '', redirectTo: 'agent/default', pathMatch: 'full' },
	{
		path: 'agent/default',
    component: LayoutComponent,
    loadChildren: './agent/agent.module#AgentModule' 
	},
	{
		path: '**',
		redirectTo: 'login'
	}
];

@NgModule({
  imports: [RouterModule.forRoot(routes)],
  exports: [RouterModule]
})
export class AppRoutingModule { }
