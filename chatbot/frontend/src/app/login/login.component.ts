import { Component, OnInit } from '@angular/core';
import { Router, ActivatedRoute } from '@angular/router';

import { AlertService, AuthenticationService, UserService } from '../_services/index';
import { User } from '../_models/index';

@Component({
    moduleId: module.id.toString(),
    templateUrl: 'login.component.html'
})

export class LoginComponent implements OnInit {
    model: any = {};
    loading = false;
    returnUrl: string;
    write: any = {"username":"test",
                  "password":"p",
                  "firstName":"a",
                  "lastName":"b"};
    users: User[] = [];

    constructor(
        private route: ActivatedRoute,
        private router: Router,
        private authenticationService: AuthenticationService,
        private userService: UserService,
        private alertService: AlertService) { }

    ngOnInit() {
        
        // reset login status
        this.authenticationService.logout();

        //clear localstroage and register the user
        localStorage.clear();
        
    }

    login() {
        this.loading = true;
        this.authenticationService.login(this.model.username, this.model.password)
            .subscribe(
                data => {
                    alert("IM IN");
                    this.router.navigate(['agent/default/intents']);
                },
                error => {
                    alert("not working");
                    this.loading = false;
                });
    }
}
