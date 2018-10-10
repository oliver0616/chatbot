import { BrowserModule } from '@angular/platform-browser';
import { NgModule } from '@angular/core';

import { AppRoutingModule } from './app-routing.module';
import { AppComponent } from './app.component';



/* Material UI imports begins here */
import {BrowserAnimationsModule} from '@angular/platform-browser/animations';
import {MatProgressSpinnerModule} from '@angular/material/progress-spinner';
/* Material UI imports ends here */


/* Project Components imports begins here */
import {CommonsModule} from './commons/commons.module'
import {DashboardModule} from './dashboard/dashboard.module';
/* Project Components imports ends here */

import { LoginComponent } from './login/index';
import { AlertService, AuthenticationService, UserService } from './_services/index';
import { HttpClientModule, HTTP_INTERCEPTORS } from '@angular/common/http';
import { JwtInterceptor } from './_helpers/index';
import { FormsModule }   from '@angular/forms';
import { AlertComponent } from './_directives/index';
import { fakeBackendProvider } from './_helpers/index';
import { AuthGuard } from './_guards/index';

@NgModule({
  imports: [
    BrowserModule,
    AppRoutingModule,
    BrowserAnimationsModule,
    CommonsModule.forRoot(),
    DashboardModule,
    MatProgressSpinnerModule,
    HttpClientModule,
    FormsModule
  ],
  declarations: [
    AppComponent,
    LoginComponent,
    AlertComponent,
  ],
  providers: [
    AuthGuard,
    AlertService,
    AuthenticationService,
    UserService,
    {
      provide: HTTP_INTERCEPTORS,
      useClass: JwtInterceptor,
      multi: true
    },
    fakeBackendProvider
  ],
  bootstrap: [AppComponent]
})
export class AppModule { }
