import { NgModule } from '@angular/core';
import { Routes, RouterModule } from '@angular/router';

import {LayoutComponent} from './dashboard/layout/layout.component'
import { LoginComponent } from './login/index';
import { AuthGuard } from './_guards';

const routes: Routes = [
	{ path: '', redirectTo: 'login', pathMatch:'full'},
	{
		path: 'login',
    component: LoginComponent,
	},
	{
		path: 'agent/default',
		component: LayoutComponent,
		canActivate:[AuthGuard],
    loadChildren: './agent/agent.module#AgentModule'
	},
	{
		path: '**',
		redirectTo: ''
	}
];

@NgModule({
  imports: [RouterModule.forRoot(routes)],
  exports: [RouterModule]
})
export class AppRoutingModule { }
