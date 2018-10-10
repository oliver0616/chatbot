import { Injectable } from '@angular/core';
import { HttpClient, HttpHeaders,HttpResponse } from '@angular/common/http';
import { Observable } from 'rxjs/Observable';
import { environment } from '../../environments/environment';
import 'rxjs/add/operator/map';

@Injectable()
export class AuthenticationService {
    constructor(private http: HttpClient) { }

    login(username: string, password: string) {
        console.log("in authentication");

        var userAuth:string = "test";
        var passAuth:string = "p";

        if(userAuth === username && passAuth === password){
            localStorage.setItem('currentUser', JSON.stringify(username));
            return <any>Observable.of(new HttpResponse({ status: 200 }));
        }
        else{
            return <any>Observable.throw("");
        }

        
    }

    logout() {
        // remove user from local storage to log user out
        localStorage.removeItem('currentUser');
    }
}  